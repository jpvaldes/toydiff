task cbuild, "Build (compile), c backend":
  switch("d", "release")
  switch("mm", "orc")
  switch("outdir", "build")
  switch("experimental", "strictEffects")
  switch("experimental", "unicodeOperators")
  switch("experimental", "overloadableEnums")
  switch("define", "nimPreviewDotLikeOps")
  switch("define", "nimPreviewFloatRoundTrip")
  switch("define", "nimStrictDelete")
  switch("hint", "[Performance]:on")
  setCommand "c"

task cdoc, "Generate documentation, c backend":
  switch("d", "release")
  switch("mm", "orc")
  switch("outdir", "doc")
  switch("experimental", "strictEffects")
  switch("experimental", "unicodeOperators")
  switch("experimental", "overloadableEnums")
  switch("define", "nimPreviewDotLikeOps")
  switch("define", "nimPreviewFloatRoundTrip")
  switch("define", "nimStrictDelete")
  switch("hint", "[Performance]:on")
  setCommand "doc"

