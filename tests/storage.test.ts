import { getStorageStatus } from '../storage';

describe('storage', () => {
  it('returns ready status', () => {
    expect(getStorageStatus()).toBe('ready');
  });
});
