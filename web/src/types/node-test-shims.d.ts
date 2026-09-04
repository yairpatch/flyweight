// The app is browser-only, so tsconfig deliberately leaves @types/node out.
// Two tests read Office fixtures off disk; these declare just what they use
// rather than pulling node's globals into the app's type surface.
declare module "node:fs" {
  export function readFileSync(path: string): Uint8Array<ArrayBuffer>;
}
declare module "node:path" {
  export function resolve(...segments: string[]): string;
}
