import { STORAGE_VERSION } from '../../storage/index';

describe('storage version', () => {
  it('exposes a semantic version string', () => {
    expect(STORAGE_VERSION).toMatch(/^\d+\.\d+\.\d+$/);
  });
});
