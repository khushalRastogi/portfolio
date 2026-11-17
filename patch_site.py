#!/usr/bin/env python3
"""
Patch for the portfolio repo. Run it from the repo root (where index.html lives).

What it does
  1. Replaces the render-blocking Google Fonts stylesheet with self-hosted, preloaded fonts (./fonts/*.woff2).
  2. Adds author and publish-date metadata (meta tags + JSON-LD) so LinkedIn Post Inspector can read them.
  3. Moves the card scan line onto transform, and stops transitioning border-color
     (the two things Lighthouse flagged as non-composited animations).
  4. Adds static contact links inside #links so crawlers that do not run JavaScript still see them.
  5. Adds a long cache header for /fonts/ to vercel.json.

Your CFG block is never touched. It is read to fill in the JSON-LD sameAs links.
Safe to run once. If anything it needs is missing it stops without changing any file.

****************************** 17/11/2026 Run Log ******************************
Patched index.html and vercel.json.
  site: https://khushal-rastogi.vercel.app/
  sameAs: ['https://www.linkedin.com/in/khushal-rastogi', 'https://github.com/khushalRastogi', 'https://huggingface.co/khushalR']
Next: git diff --stat, open the page locally, then commit and push.
********************************************************************************


"""
import datetime
import json
import pathlib
import re
import sys

MARK = "<!-- patch-2026-11-17 -->"
PUBLISHED = "2026-11-17"

root = pathlib.Path.cwd()
index_p = root / "index.html"
vercel_p = root / "vercel.json"


def die(msg):
    print("STOPPED, nothing was changed:", msg)
    sys.exit(1)


def read_text(path):
    with path.open("r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write_text(path, text):
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)


if not index_p.exists():
    die("index.html not found. Run this from the repo root.")
for f in ("bricolage-latin.woff2", "newsreader-latin.woff2"):
    if not (root / "fonts" / f).exists():
        die("fonts/%s not found. Copy the fonts folder into the repo root first." % f)

html = read_text(index_p)
if MARK in html:
    die("this patch is already applied.")


def replace_once(old, new, label):
    global html
    n = html.count(old)
    if n != 1:
        die("expected 1 match for '%s' but found %d." % (label, n))
    html = html.replace(old, new, 1)


# ---- read what we need from the existing page --------------------------------
m = re.search(r'<link rel="canonical" href="([^"]+)"', html)
if not m:
    die("canonical link not found.")
site = m.group(1)
if "YOUR-SITE" in site:
    die("canonical still says YOUR-SITE. Replace the placeholder first.")
site = site.rstrip("/") + "/"

cfg_line = re.search(r"const CFG=\{.*?\};", html, re.S)
if not cfg_line:
    die("CFG block not found.")
cfg_text = cfg_line.group(0)


def cfg(key):
    mm = re.search(r"\b%s\s*:\s*([\"'])(.*?)\1" % key, cfg_text)
    val = mm.group(2).strip() if mm else ""
    if val and not re.match(r"^(https?:|mailto:)", val):
        val = site + val.lstrip("/")
    return val


linkedin, github, hf, blog, resume = (cfg(k) for k in ("linkedin", "github", "hf", "blog", "resume"))
same_as = [u for u in (linkedin, github, hf) if u]

# ---- 1. fonts ----------------------------------------------------------------
html, n = re.subn(
    r'<link rel="stylesheet" href="https://fonts\.googleapis\.com/[^"]*">[ \t]*\r?\n?',
    '<link rel="preload" href="fonts/bricolage-latin.woff2" as="font" type="font/woff2" crossorigin>\n'
    '<link rel="preload" href="fonts/newsreader-latin.woff2" as="font" type="font/woff2" crossorigin>\n',
    html,
)
if n != 1:
    die("Google Fonts stylesheet link not found exactly once (found %d)." % n)

replace_once(
    "<style>",
    "<style>\n"
    '@font-face{font-family:"Bricolage Grotesque";src:url(fonts/bricolage-latin.woff2) format("woff2");'
    "font-weight:400 800;font-style:normal;font-display:swap}\n"
    '@font-face{font-family:"Newsreader";src:url(fonts/newsreader-latin.woff2) format("woff2");'
    "font-weight:400 500;font-style:normal;font-display:swap}",
    "<style>",
)

# ---- 3. animations -----------------------------------------------------------
replace_once(
    '.doc.scan::before{content:"";position:absolute;left:0;right:0;top:0;height:2px;background:var(--stamp);'
    "box-shadow:0 0 14px 2px var(--stamp);animation:sc 2.4s linear}",
    '.doc.scan::before{content:"";position:absolute;left:0;right:0;top:0;height:100%;pointer-events:none;'
    "background:linear-gradient(var(--stamp),var(--stamp)) 0 0/100% 2px no-repeat,"
    "linear-gradient(to bottom,color-mix(in srgb,var(--stamp) 28%,transparent),transparent) 0 2px/100% 22px no-repeat;"
    "animation:sc 2.4s linear}",
    "scan line rule",
)
replace_once(
    "@keyframes sc{from{top:0}to{top:100%}}",
    "@keyframes sc{from{transform:translateY(-4px)}to{transform:translateY(100%)}}",
    "scan keyframes",
)
replace_once(
    "transition:border-color .25s,background .35s,color .35s",
    "transition:background .35s,color .35s",
    "field transition",
)
# stops the headline leaving a single word ("it.") alone on its last line
replace_once(
    "font-size:clamp(2.1rem,5.6vw,3.4rem)",
    "font-size:clamp(2.1rem,5.6vw,3.4rem);text-wrap:balance",
    "h1 size",
)

# ---- 2. metadata + JSON-LD ---------------------------------------------------
today = datetime.date.today().isoformat()
meta = (
    "\n" + MARK + "\n"
    '<meta name="author" content="Khushal Rastogi">\n'
    '<meta property="article:author" content="%s">\n'
    '<meta property="article:published_time" content="%s">\n' % (linkedin or site, PUBLISHED)
)
replace_once('<meta name="theme-color" content="#0f1b2d">', '<meta name="theme-color" content="#0f1b2d">' + meta.rstrip("\n"), "theme-color")

person = {
    "@type": "Person",
    "name": "Khushal Rastogi",
    "jobTitle": "Senior Data Scientist",
    "url": site,
    "worksFor": {"@type": "Organization", "name": "Newgen Software"},
    "alumniOf": {"@type": "CollegeOrUniversity", "name": "University of Delhi"},
    "knowsAbout": ["Document AI", "Retrieval-augmented generation", "Agentic AI systems", "Computer vision", "Machine learning"],
}
if same_as:
    person["sameAs"] = same_as
ld = {
    "@context": "https://schema.org",
    "@type": "ProfilePage",
    "url": site,
    "name": "Khushal Rastogi: Senior Data Scientist",
    "datePublished": PUBLISHED,
    "dateModified": today,
    "author": person,
    "mainEntity": person,
}
ld_tag = '<script type="application/ld+json">' + json.dumps(ld, ensure_ascii=False).replace("</", "<\\/") + "</script>\n"
replace_once("</head>", ld_tag + "</head>", "</head>")

# ---- 4. static contact links -------------------------------------------------
links = [("LinkedIn", linkedin), ("GitHub", github), ("Hugging Face", hf), ("Writing", blog), ("Resume", resume)]
static = "".join(
    '<a href="%s"%s>%s</a>' % (u, ' target="_blank" rel="noopener"' if u.startswith("http") else "", n)
    for n, u in links
    if u
)
replace_once('<div id="links"></div>', '<div id="links">%s</div>' % static, "#links")

# ---- 5. vercel.json ----------------------------------------------------------
vj = {}
if vercel_p.exists():
    try:
        with vercel_p.open("r", encoding="utf-8") as fh:
            vj = json.loads(fh.read())
    except ValueError:
        die("vercel.json is not valid JSON.")
rules = vj.setdefault("headers", [])
if not any(r.get("source") == "/fonts/(.*)" for r in rules):
    rules.append({"source": "/fonts/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=31536000, immutable"}]})
vj.setdefault("cleanUrls", True)

write_text(index_p, html)
with vercel_p.open("w", encoding="utf-8", newline="") as fh:
    fh.write(json.dumps(vj, indent=2) + "\n")
print("Patched index.html and vercel.json.")
print("  site:", site)
print("  sameAs:", same_as or "(only the page itself; add github/hf to CFG and re-run on a fresh copy if needed)")
print("Next: git diff --stat, open the page locally, then commit and push.")
