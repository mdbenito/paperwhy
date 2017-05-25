# Paperwhy

This is PaperWhy. Our sisyphean endeavour not to drown in the
immense Machine Learning literature.

With thousands of papers every month, keeping up with and making sense
of recent research in machine learning has become almost
impossible. By routinely reviewing and reporting papers we help
ourselves and hopefully someone else.

## Setup

In order to preview the site you need to download and
install [Hugo](gohugo.io). From the source of the repo type

```
hugo server
```

and open a browser to [localhost:1313](//localhost:1313).

## Adding a new post

Paths here are relative to the root of the code.

1. Add an entry to the Zotero database for the paper and any papers
   that you are going to cite in the post.
2. Export the database to bibtex, then convert it to yaml with
   `pandoc-citeproc -y bibtexfile > data/bibliography.yml`.
3. Posts are stored in `content/post/` as markdown files.

The **extra** field in Zotero (**notes** in the bibtex) can hold
further fields. For now only `code: http://urltowhatever` (for any
sourcecode published with the paper) is used in the templates to
automatically add a link to it next to the one to the original paper,
authors, etc.

## Structure of a post

Each markdown file has a header, the **frontmatter**, with
([easy-hugo for emacs]() copies it from `archetypes/default.md`):

```
author: Your full name here
authors: ["Repeat your full name here, I have to fix this"]
tags: [""]

paper_authors: ["surname, name", "surname, name"]
paper_key: "bibtex_citekey"
```

Even though they can be extracted from the bibliography file, it is
necessary to copy the authors to `paper_authors`. This is required
for the "Papers by..." pages.

The name of the file is in principle arbitrary, but we might want to
use the bibtex citekey and drop the variable in the frontmatter.

**Citations** in the markdown are done with
a [Hugo shortcode](gohugo.io/extras/shortcodes/) as follows:

```
This architecture was already tested in {{< cite bibtex_citekey_here>}}
```

This is replaced either by a standard "Paper title, Authors, Year", or
possibly by a number and footnote, or in case the paper was the
subject of a previous post by a link to it.

**Videos** can be embedded with another shortcode:

```
{{< youtube videoid >}}
```

**Figures** can be added with
```
{{< figure src="/img/citekey-fig1.svg" title="some caption" >}}
```
There's also a caption field, but looks worse in the current theme.

## To do

* Fix the generation of the post author's pages. `{{ . Content }}` is
  being ignored in the template. Should I place the `_index.md`
  elsewhere?

* Generate lists of authors (and citekeys if I use them) upon each
  build. Use them when writing new posts, maybe autocomplete in
  `easy-hugo`?
  
* Consolidate multiple assets in single files.

* Ensure that all content is minified by aerobatic before being served
  and in case it doesn't write a Makefile.
  
* Use local, stripped version of MathJax?

## Meta

This site is built with [HUGO](gohugo.io) based on
the [hikari](github.com/digitalcraftsman/hugo-hikari-theme) theme.
Deployment is through bitbucket pipelines
via [aerobatic](aerobatic.com).  A custom docker image is used because
we need Hugo >= 0.20 and currently (May 2017) aerobatic's docker
container for hugo has version 0.19

## Credits

* The many authors of the papers
* HUGO
* hikari theme (port and original author)
* MathJax
* jQuery
* icomoon
* ...
