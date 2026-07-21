export async function resolve(specifier, context, next) {
  if ((specifier.startsWith("./") || specifier.startsWith("../")) && !/\.[a-z]+$/i.test(specifier)) {
    try {
      return await next(specifier + ".js", context)
    } catch {
      // fall through to the default resolution for a real error message
    }
  }
  return next(specifier, context)
}
