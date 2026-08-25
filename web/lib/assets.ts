/**
 * Where the demo models, motion and music come from.
 *
 * A deployed build reads them from R2, whose egress is free, so the ~43MB a
 * visitor downloads never touches the deployment's transfer budget — one pool
 * shared across every project on the account. `next dev` reads the same files
 * out of `public/`, which keeps a checkout self-contained: drop in a model,
 * reload, no round trip through a bucket.
 *
 * Keys there are versioned by path, which is what lets them carry a one-year
 * immutable cache header: rename, never overwrite in place.
 */
export const ASSETS = process.env.NODE_ENV === "production" ? "https://assets.reze.one/demo/reze-engine" : ""
