import { randomBytes } from 'node:crypto';
import { execFile } from 'node:child_process';
import { mkdir, realpath, rm, stat, writeFile } from 'node:fs/promises';
import { isAbsolute, join } from 'node:path';
import { promisify } from 'node:util';

export type MediaTool = 'ffmpeg' | 'ffprobe';
export function assertMediaTool(value: unknown): MediaTool {
  if (value !== 'ffmpeg' && value !== 'ffprobe') throw new Error('Invalid media tool');
  return value;
}

async function writeAuthorization(dataDir: string, kind: string, path: string) {
  await mkdir(dataDir, { recursive: true, mode: 0o700 });
  const directory = join(await realpath(dataDir), '.path-authorizations');
  await mkdir(directory, { recursive: true, mode: 0o700 });
  const authorization = randomBytes(32).toString('hex');
  await writeFile(
    join(directory, authorization + '.json'),
    JSON.stringify({ token: authorization, kind, path }),
    { flag: 'wx', mode: 0o600 },
  );
  return { authorization, path };
}

export async function authorizeMediaTool(dataDir: string, selected: string, kind: MediaTool) {
  assertMediaTool(kind);
  if (!isAbsolute(dataDir) || !isAbsolute(selected)) throw new Error('Expected absolute paths');
  const path = await realpath(selected);
  if (!(await stat(path)).isFile()) throw new Error('Expected an executable file');
  const { stdout, stderr } = await promisify(execFile)(path, ['-version'], {
    windowsHide: true,
    timeout: 5000,
    maxBuffer: 128 * 1024,
  });
  if (!(stdout + stderr).trimStart().startsWith(kind + ' version ')) {
    throw new Error('Selected executable does not match the media tool');
  }
  return writeAuthorization(dataDir, kind, path);
}

export async function authorizeModelsDirectory(dataDir: string, selected: string) {
  if (!isAbsolute(dataDir)) throw new Error('Expected an absolute data directory');
  if (selected === '') return writeAuthorization(dataDir, 'models_dir', '');
  if (!isAbsolute(selected)) throw new Error('Expected an absolute models directory');

  await mkdir(selected, { recursive: true });
  const path = await realpath(selected);
  if (!(await stat(path)).isDirectory()) throw new Error('Expected a directory');
  const probe = join(path, `.voicestudio-write-test-${randomBytes(8).toString('hex')}`);
  try {
    await writeFile(probe, 'ok', { flag: 'wx', mode: 0o600 });
  } catch (error) {
    throw new Error('Directory is not writable', { cause: error });
  } finally {
    await rm(probe, { force: true }).catch(() => undefined);
  }
  return writeAuthorization(dataDir, 'models_dir', path);
}

export type VoiceConversionDirectoryKind = 'applio_dir' | 'rvc_dataset_dir';
export function assertVoiceConversionDirectoryKind(value: unknown): VoiceConversionDirectoryKind {
  if (value !== 'applio_dir' && value !== 'rvc_dataset_dir') {
    throw new Error('Invalid voice-conversion directory kind');
  }
  return value;
}

async function isFile(path: string) {
  try {
    return (await stat(path)).isFile();
  } catch {
    return false;
  }
}

/**
 * Mint a one-shot capability for an Applio checkout (must contain core.py and
 * rvc/infer/infer.py; '' restores the managed default) or an RVC training
 * dataset folder (read-only; never created or written).
 */
export async function authorizeVoiceConversionDirectory(
  dataDir: string,
  selected: string,
  kind: VoiceConversionDirectoryKind,
) {
  assertVoiceConversionDirectoryKind(kind);
  if (!isAbsolute(dataDir)) throw new Error('Expected an absolute data directory');
  if (selected === '' && kind === 'applio_dir') return writeAuthorization(dataDir, kind, '');
  if (!isAbsolute(selected)) throw new Error('Expected an absolute directory');
  const path = await realpath(selected);
  if (!(await stat(path)).isDirectory()) throw new Error('Expected a directory');
  if (
    kind === 'applio_dir' &&
    !((await isFile(join(path, 'core.py'))) && (await isFile(join(path, 'rvc', 'infer', 'infer.py'))))
  ) {
    throw new Error('Selected folder is not an Applio checkout');
  }
  return writeAuthorization(dataDir, kind, path);
}
