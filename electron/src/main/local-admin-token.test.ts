import { describe, expect, it } from 'vitest';
import { proxyRequestHeaders } from './dev-backend-proxy';
import { LOCAL_ADMIN_TOKEN_HEADER, localAdminToken } from './local-admin-token';

describe('local admin token', () => {
  it('is a stable 256-bit per-launch secret', () => {
    expect(localAdminToken()).toMatch(/^[A-Za-z0-9_-]{43}$/);
    expect(localAdminToken()).toBe(localAdminToken());
  });

  it('cannot be forged by the renderer through the dev proxy', () => {
    const headers = proxyRequestHeaders(
      { 'x-voicestudio-local-token': 'forged', accept: 'application/json' },
      { [LOCAL_ADMIN_TOKEN_HEADER]: localAdminToken() },
    );
    expect(headers['x-voicestudio-local-token']).toBeUndefined();
    expect(headers[LOCAL_ADMIN_TOKEN_HEADER]).toBe(localAdminToken());
    expect(headers.accept).toBe('application/json');
  });
});
