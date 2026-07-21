// dist/ is emitted with extensionless relative imports (moduleResolution: bundler —
// the web app's bundler resolves them). This hook lets plain `node --test` import
// dist by retrying relative specifiers with ".js" appended.
import { register } from "node:module"

register("./resolve-ext.mjs", import.meta.url)
