# Third-party code

This folder is called `lib/`, not `vendor/`, on purpose: the site's `.gitignore` and `_config.yml` both exclude
anything named `vendor` (for Ruby gems), which would keep these files out of git and out of the published site.

`three.module.min.js` is three.js 0.186.0 (MIT, see `THREE-LICENSE`), vendored so the game has no CDN
dependency and works offline. The npm package no longer ships a minified build, so this file is
`build/three.module.js` and `build/three.core.js` bundled into one ES module and minified:

    npm pack three@0.186.0 && tar -xzf three-0.186.0.tgz
    npx esbuild@0.25.0 package/build/three.module.js --bundle --format=esm --minify --legal-comments=inline --outfile=three.module.min.js

To upgrade, repeat with the new version and update this note.
