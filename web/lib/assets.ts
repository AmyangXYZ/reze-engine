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

/** The cast, shared by every site (reze.design reads the same Reze) rather than
 *  copied into each one's folder. `next dev` reads the same models out of
 *  `public/models/reze`.
 *
 *  `reze-webp`: Reze and her bomb form with their textures as WebP (1.9MB where
 *  the PNGs were 16.3MB). */
export const CAST = process.env.NODE_ENV === "production" ? "https://assets.reze.one/demo/reze-webp" : "/models/reze"
