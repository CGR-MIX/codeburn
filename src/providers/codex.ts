import { readdir, readFile, stat } from 'fs/promises'
import { basename, join } from 'path'
import { homedir } from 'os'

import { extractBashCommands } from '../bash-utils.js'
import { calculateCost } from '../models.js'
import type { Provider, SessionSource, SessionParser, ParsedProviderCall } from './types.js'

const modelDisplayNames: Record<string, string> = {
  'gpt-5.3-codex': 'GPT-5.3 Codex',
  'gpt-5.4-mini': 'GPT-5.4 Mini',
  'gpt-5.4': 'GPT-5.4',
  'gpt-5': 'GPT-5',
  'gpt-4o-mini': 'GPT-4o Mini',
  'gpt-4o': 'GPT-4o',
}

const toolNameMap: Record<string, string> = {
  exec_command: 'Bash',
  shell_command: 'Bash',
  read_file: 'Read',
  write_file: 'Edit',
  apply_diff: 'Edit',
  apply_patch: 'Edit',
  spawn_agent: 'Agent',
  close_agent: 'Agent',
  wait_agent: 'Agent',
  read_dir: 'Glob',
}

type CodexEntry = {
  type: string
  timestamp?: string
  payload?: {
    type?: string
    role?: string
    cwd?: string
    model_provider?: string
    originator?: string
    id?: string
    session_id?: string
    model?: string
    name?: string
    arguments?: string | Record<string, unknown>
    content?: Array<{ type?: string; text?: string }>
    info?: {
      model?: string
      model_name?: string
      last_token_usage?: CodexTokenUsage
      total_token_usage?: CodexTokenUsage
    }
  }
}

type CodexTokenUsage = {
  input_tokens?: number
  cached_input_tokens?: number
  output_tokens?: number
  reasoning_output_tokens?: number
  total_tokens?: number
}

function getCodexDir(override?: string): string {
  return override ?? process.env['CODEX_HOME'] ?? join(homedir(), '.codex')
}

function sanitizeProject(cwd: string): string {
  return cwd.replace(/^[A-Za-z]:[\\/]/, '').replace(/^[/\\]/, '').replace(/[\\/]/g, '-')
}

async function readFirstLine(filePath: string): Promise<CodexEntry | null> {
  try {
    const content = await readFile(filePath, 'utf-8')
    const line = content.split('\n')[0]
    if (!line?.trim()) return null
    return JSON.parse(line) as CodexEntry
  } catch {
    return null
  }
}

async function isValidCodexSession(filePath: string): Promise<{ valid: boolean; meta?: CodexEntry }> {
  const entry = await readFirstLine(filePath)
  if (!entry) return { valid: false }
  const valid = entry.type === 'session_meta' &&
    typeof entry.payload?.originator === 'string' &&
    entry.payload.originator.toLowerCase().startsWith('codex')
  return { valid, meta: valid ? entry : undefined }
}

async function addSessionSource(filePath: string, sources: SessionSource[], seenSessionKeys: Set<string>) {
  const s = await stat(filePath).catch(() => null)
  if (!s?.isFile()) return

  const { valid, meta } = await isValidCodexSession(filePath)
  if (!valid || !meta) return

  const sessionKey = meta.payload?.session_id ?? meta.payload?.id ?? filePath
  if (seenSessionKeys.has(sessionKey)) return
  seenSessionKeys.add(sessionKey)

  const cwd = meta.payload?.cwd ?? 'unknown'
  sources.push({ path: filePath, project: sanitizeProject(cwd), provider: 'codex' })
}

async function discoverFlatSessionDir(dir: string, sources: SessionSource[], seenSessionKeys: Set<string>) {
  const files = await readdir(dir).catch(() => [] as string[])
  for (const file of files) {
    if (!file.startsWith('rollout-') || !file.endsWith('.jsonl')) continue
    await addSessionSource(join(dir, file), sources, seenSessionKeys)
  }
}

async function discoverSessionsInDir(codexDir: string): Promise<SessionSource[]> {
  const sessionsDir = join(codexDir, 'sessions')
  const sources: SessionSource[] = []
  const seenSessionKeys = new Set<string>()

  const years = await readdir(sessionsDir).catch(() => [] as string[])

  for (const year of years) {
    if (!/^\d{4}$/.test(year)) continue
    const yearDir = join(sessionsDir, year)
    const months = await readdir(yearDir).catch(() => [] as string[])

    for (const month of months) {
      if (!/^\d{2}$/.test(month)) continue
      const monthDir = join(yearDir, month)
      const days = await readdir(monthDir).catch(() => [] as string[])

      for (const day of days) {
        if (!/^\d{2}$/.test(day)) continue
        const dayDir = join(monthDir, day)
        const files = await readdir(dayDir).catch(() => [] as string[])

        for (const file of files) {
          if (!file.startsWith('rollout-') || !file.endsWith('.jsonl')) continue
          const filePath = join(dayDir, file)
          await addSessionSource(filePath, sources, seenSessionKeys)
        }
      }
    }
  }

  await discoverFlatSessionDir(join(codexDir, 'archived_sessions'), sources, seenSessionKeys)

  return sources
}

function resolveModel(info: CodexEntry['payload'], sessionModel?: string): string {
  return info?.info?.model
    ?? info?.info?.model_name
    ?? sessionModel
    ?? 'gpt-5'
}

function parseFunctionArguments(args: string | Record<string, unknown> | undefined): Record<string, unknown> | null {
  if (!args) return null
  if (typeof args !== 'string') return args

  try {
    const parsed = JSON.parse(args) as unknown
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : null
  } catch {
    return null
  }
}

function extractShellCommands(payload: CodexEntry['payload']): string[] {
  const args = parseFunctionArguments(payload?.arguments)
  const command = args?.command
  return typeof command === 'string' ? extractBashCommands(command) : []
}

function createParser(source: SessionSource, seenKeys: Set<string>): SessionParser {
  return {
    async *parse(): AsyncGenerator<ParsedProviderCall> {
      let content: string
      try {
        content = await readFile(source.path, 'utf-8')
      } catch {
        return
      }

      const lines = content.split('\n').filter(l => l.trim())
      let sessionModel: string | undefined
      let sessionId = ''
      let prevCumulativeTotal = 0
      let prevInput = 0
      let prevCached = 0
      let prevOutput = 0
      let prevReasoning = 0
      let pendingTools: string[] = []
      let pendingBashCommands: string[] = []
      let pendingUserMessage = ''

      for (const line of lines) {
        let entry: CodexEntry
        try {
          entry = JSON.parse(line) as CodexEntry
        } catch {
          continue
        }

        if (entry.type === 'session_meta') {
          sessionId = entry.payload?.session_id ?? entry.payload?.id ?? basename(source.path, '.jsonl')
          sessionModel = entry.payload?.model
          continue
        }

        if (entry.type === 'response_item' && entry.payload?.type === 'function_call') {
          const rawName = entry.payload.name ?? ''
          pendingTools.push(toolNameMap[rawName] ?? rawName)
          if (toolNameMap[rawName] === 'Bash') {
            pendingBashCommands.push(...extractShellCommands(entry.payload))
          }
          continue
        }

        if (entry.type === 'response_item' && entry.payload?.type === 'message' && entry.payload?.role === 'user') {
          const texts = (entry.payload.content ?? [])
            .filter(c => c.type === 'input_text')
            .map(c => c.text ?? '')
            .filter(Boolean)
          if (texts.length > 0) pendingUserMessage = texts.join(' ')
          continue
        }

        if (entry.type === 'event_msg' && entry.payload?.type === 'token_count') {
          const info = entry.payload.info
          if (!info) continue

          const cumulativeTotal = info.total_token_usage?.total_tokens ?? 0
          if (cumulativeTotal > 0 && cumulativeTotal === prevCumulativeTotal) continue
          prevCumulativeTotal = cumulativeTotal

          const last = info.last_token_usage
          let inputTokens = 0
          let cachedInputTokens = 0
          let outputTokens = 0
          let reasoningTokens = 0

          if (last) {
            inputTokens = last.input_tokens ?? 0
            cachedInputTokens = last.cached_input_tokens ?? 0
            outputTokens = last.output_tokens ?? 0
            reasoningTokens = last.reasoning_output_tokens ?? 0
          } else if (cumulativeTotal > 0) {
            const total = info.total_token_usage
            if (!total) continue
            inputTokens = (total.input_tokens ?? 0) - prevInput
            cachedInputTokens = (total.cached_input_tokens ?? 0) - prevCached
            outputTokens = (total.output_tokens ?? 0) - prevOutput
            reasoningTokens = (total.reasoning_output_tokens ?? 0) - prevReasoning
          }

          if (!last) {
            const total = info.total_token_usage
            if (total) {
              prevInput = total.input_tokens ?? 0
              prevCached = total.cached_input_tokens ?? 0
              prevOutput = total.output_tokens ?? 0
              prevReasoning = total.reasoning_output_tokens ?? 0
            }
          }

          const totalTokens = inputTokens + cachedInputTokens + outputTokens + reasoningTokens
          if (totalTokens === 0) continue

          // OpenAI includes cached tokens inside input_tokens; Anthropic does not.
          // Normalize to Anthropic semantics: inputTokens = non-cached only.
          const uncachedInputTokens = Math.max(0, inputTokens - cachedInputTokens)

          const model = resolveModel(entry.payload, sessionModel)
          const timestamp = entry.timestamp ?? ''
          const dedupKey = `codex:${source.path}:${timestamp}:${cumulativeTotal}`

          if (seenKeys.has(dedupKey)) continue
          seenKeys.add(dedupKey)

          const costUSD = calculateCost(
            model,
            uncachedInputTokens,
            outputTokens + reasoningTokens,
            0,
            cachedInputTokens,
            0,
          )

          yield {
            provider: 'codex',
            model,
            inputTokens: uncachedInputTokens,
            outputTokens,
            cacheCreationInputTokens: 0,
            cacheReadInputTokens: cachedInputTokens,
            cachedInputTokens,
            reasoningTokens,
            webSearchRequests: 0,
            costUSD,
            tools: pendingTools,
            bashCommands: pendingBashCommands,
            timestamp,
            speed: 'standard',
            deduplicationKey: dedupKey,
            userMessage: pendingUserMessage,
            sessionId,
          }

          pendingTools = []
          pendingBashCommands = []
          pendingUserMessage = ''
        }
      }
    },
  }
}

export function createCodexProvider(codexDir?: string): Provider {
  const dir = getCodexDir(codexDir)

  return {
    name: 'codex',
    displayName: 'Codex',

    modelDisplayName(model: string): string {
      for (const [key, name] of Object.entries(modelDisplayNames)) {
        if (model.startsWith(key)) return name
      }
      return model
    },

    toolDisplayName(rawTool: string): string {
      return toolNameMap[rawTool] ?? rawTool
    },

    async discoverSessions(): Promise<SessionSource[]> {
      return discoverSessionsInDir(dir)
    },

    createSessionParser(source: SessionSource, seenKeys: Set<string>): SessionParser {
      return createParser(source, seenKeys)
    },
  }
}

export const codex = createCodexProvider()
