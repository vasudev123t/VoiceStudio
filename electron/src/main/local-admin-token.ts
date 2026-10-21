import { randomBytes } from 'node:crypto';

/**
 * Per-launch secret shared by the Electron main process and the backend it
 * spawns. The backend requires it on admin routes even from loopback, so a
 * random local process can no longer call `/system/set-env`, installers or
 * `/voice-conversion/setup`. It is passed to the backend in the environment
 * (the backend removes it from its own env at startup), injected by main into
 * every proxied request, and never exposed to the renderer.
 */
export const LOCAL_ADMIN_TOKEN_ENV = 'OMNIVOICE_LOCAL_ADMIN_TOKEN';
export const LOCAL_ADMIN_TOKEN_HEADER = 'X-VoiceStudio-Local-Token';

let token: string | null = null;

export function localAdminToken(): string {
  token ??= randomBytes(32).toString('base64url');
  return token;
}
