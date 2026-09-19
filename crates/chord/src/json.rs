//! JSON validation restated for owned data, ported from upstream
//! `src/json.ts`: the acceptance surface `isJsonValue` checks for the owned
//! `JsonValue` tree (finite numbers only, no cycles, no exotic prototypes,
//! no sparse arrays), a `JsonNumber` constructor surface that keeps
//! non-finite values unrepresentable, and the helpers the `json.test.ts`
//! port drives. Upstream caps recursion depth at 512 to bound hostile
//! payloads; the same cap applies to the owned tree as a wire contract.
