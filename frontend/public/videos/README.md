# Higgsfield video assets

Drop your Higgsfield.ai exports into this folder with these exact filenames.
The site auto-detects them — if a file is missing, a code-driven animation
(particle globe / gradient) renders instead, so the page never breaks.

| Filename | Where it plays | Suggested Higgsfield prompt | Specs |
|---|---|---|---|
| `hero-earth.mp4` | Landing hero background (full screen) | "Cinematic slow orbit around planet Earth from space at night, green aurora data grid lines sweeping across continents, satellites drifting past, dark atmosphere, photorealistic, slow dolly motion" | 1920×1080, 8–12s seamless loop, no audio, H.264, < 8 MB |
| `feature-deforestation.mp4` | Deforestation feature card | "Aerial satellite timelapse of a rainforest canopy, subtle camera push-in, patches highlighted with soft green scanning overlay, moody cinematic light" | 1280×720, 5–8s loop, < 4 MB |
| `feature-flooding.mp4` | Flooding feature card | "Top-down satellite view of a river delta flooding, water spreading in slow motion, teal radar scan sweep effect, cinematic" | 1280×720, 5–8s loop, < 4 MB |
| `feature-ice.mp4` | Ice melt feature card | "Arctic glacier from orbit, ice sheet slowly calving, cold blue tones with subtle green grid overlay, slow aerial drift" | 1280×720, 5–8s loop, < 4 MB |
| `auth-ambient.mp4` | Sign-in / sign-up brand panel | "Abstract slow-motion Earth horizon from orbit at dawn, dark green and black palette, soft particles, minimal, meditative loop" | 1080×1920 or 1080×1080, 8–12s loop, < 6 MB |

Tips:
- Export "loop" mode in Higgsfield so the last frame matches the first.
- Compress before committing: `ffmpeg -i in.mp4 -vcodec libx264 -crf 28 -an out.mp4`
- Keep total video weight under ~25 MB or the page will feel slow on 4G.
- Poster frames (optional): same name with `.jpg` (e.g. `hero-earth.jpg`).

Image sources: use your own Higgsfield stills, Unsplash/Pexels (free license),
or NASA/ESA imagery (public domain). Do NOT ship images saved from Pinterest —
they are not licensed for reuse; use Pinterest for mood-boarding only.
