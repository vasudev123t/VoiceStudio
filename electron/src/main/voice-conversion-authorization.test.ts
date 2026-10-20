import { mkdir, mkdtemp, readFile, realpath, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, expect, it } from 'vitest';
import {
  assertVoiceConversionDirectoryKind,
  authorizeVoiceConversionDirectory,
} from './media-authorization';

const roots: string[] = [];
afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

async function scratch(prefix: string) {
  const root = await mkdtemp(join(tmpdir(), prefix));
  roots.push(root);
  return root;
}

async function payload(dataDir: string, token: string) {
  return JSON.parse(
    await readFile(join(dataDir, '.path-authorizations', `${token}.json`), 'utf8'),
  );
}

it('authorizes a real Applio checkout and refuses other folders', async () => {
  const root = await scratch('voicestudio-applio-');
  const dataDir = join(root, 'data');
  const applio = join(root, 'Applio');
  await mkdir(join(applio, 'rvc', 'infer'), { recursive: true });
  await writeFile(join(applio, 'core.py'), '');
  await writeFile(join(applio, 'rvc', 'infer', 'infer.py'), '');

  const result = await authorizeVoiceConversionDirectory(dataDir, applio, 'applio_dir');
  expect(await payload(dataDir, result.authorization)).toEqual({
    token: result.authorization,
    kind: 'applio_dir',
    path: await realpath(applio),
  });

  await expect(authorizeVoiceConversionDirectory(dataDir, root, 'applio_dir')).rejects.toThrow(
    /not an Applio checkout/,
  );
});

it('authorizes a dataset folder without writing into it, and resets Applio with an empty path', async () => {
  const root = await scratch('voicestudio-rvc-dataset-');
  const dataDir = join(root, 'data');
  const dataset = join(root, 'my voice');
  await mkdir(dataset);

  const result = await authorizeVoiceConversionDirectory(dataDir, dataset, 'rvc_dataset_dir');
  expect((await payload(dataDir, result.authorization)).kind).toBe('rvc_dataset_dir');

  const reset = await authorizeVoiceConversionDirectory(dataDir, '', 'applio_dir');
  expect(reset.path).toBe('');
  await expect(
    authorizeVoiceConversionDirectory(dataDir, '', 'rvc_dataset_dir'),
  ).rejects.toThrow();
  await expect(
    authorizeVoiceConversionDirectory(dataDir, join(root, 'missing'), 'rvc_dataset_dir'),
  ).rejects.toThrow();
});

it('rejects unknown capability kinds', () => {
  expect(() => assertVoiceConversionDirectoryKind('models_dir')).toThrow();
  expect(assertVoiceConversionDirectoryKind('applio_dir')).toBe('applio_dir');
});
