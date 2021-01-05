# Paperwhy

This is PaperWhy. Our sisyphean endeavour not to drown in the immense
Machine Learning literature.

With thousands of papers every month, keeping up with and making sense
of recent research in machine learning has become almost
impossible. By routinely reviewing and reporting papers we help
ourselves and hopefully someone else.

## Requirements and setup

* [pandoc](http://pandoc.org/) with `pandoc-citeproc` to generate the
  bibliography.
* [Hugo](https://gohugo.io) **version 0.41** to locally generate the site.
  Breaking changes after this version make upgrading too much of a hassle.
* Optional: [TeXmacs](http://www.texmacs.org) with
  the [markdown converter](https://bitbucket.org/mdbenito/tm2md) to
  write the posts in a sensible editor with typesetting.

In order to preview the site, from the source of the repo type

```
hugo server
```

and open a browser to [localhost:1313](//localhost:1313).

All original content is (optionally) written as TeXmacs files inside
`tmdocs/`, then exported to markdown using our
TeXmacs2Markdown [converter](https://bitbucket.org/mdbenito/tm2md). If
this hasn't been pushed to TeXmacs' trunk it can be easily installed
following the instructions at the projects page. Directly writing
markdown is of course also possible, see below.

### Using docker

Alternatively to installing hugo, you can use the provided Dockerfile to first
build an image:

```shell
docker build -f docker/Dockerfile -t paperwhy .
```

and then run hugo from a container with:

```shell
docker run --rm -it -v $(pwd):/site -p 1313:1313 paperwhy \
       serve --bind 0.0.0.0
```

## Adding a new post using TeXmacs

Paths are relative to the root of the code.
 
1. Write the original document in `tmdocs/`. The naming convention is
   to use the bibtex citekey for the paper as basename, e.g.
   `turing_chemical_1952.tm`. There is a TeXmacs style file with some
   default settings. See also below for how to set things up to get
   metadata automatically exported to markdown.
1. Save any images inside `static/img/`.
1. Add an entry to the [Zotero database]() for the paper and any
   others that you are going to cite in the post. Export the database
   to bibtex as `tmdocs/paperwhy.bib` and cite as usual inside the
   TeXmacs document.
1. Convert the bibtex to yaml with
   `pandoc-citeproc -y tmdocs/paperwhy.bib > data/bibliography.yml`.
1. Export your posts as markdown into `content/post/`.
1. **Two things need manual fixing in the markdown for now:** the
   `paper_key` field and paths to images, which need to be corrected
   to `/img/whatever`.

The **extra** field in Zotero (**notes** in the bibtex) can hold
further fields. For now only `code: http://urltowhatever` (for any
source code published with the paper) is used in the templates to
automatically add a link to it next to the one to the original paper,
authors, etc.

### Supported Hugo features for TeXmacs documents

* Paper authors: use the doc-data author tags as if adding a regular
  author to the document.
* Post author: use the running-author macro.
* Tags: use the \tags macro.
* Shortcodes: use the \hugo-shortcode macro. To do: use xargs to
  accept a variable number of arguments.
* Bibliography: Add the bibliography file and cite as usual.
* Almost everything that Hugo supports can be converted from TeXmacs,
  including all kinds of text, lists, images, links and blackfriday
  extensions like footnotes and ~~striked through text~~.
* Furthermore, much of TeXmacs' non-dynamic markup is recognized and
  exported. In particular, labels and references, numbered
  environments, and figures should work out of the box.

## Structure of a markdown post

Each markdown file has a header, the **frontmatter**, with:

```
author: Your full name here
authors: ["Repeat your full name here, I have to fix this"]
date: YYYY-mm-dd
tags: ["tags", "should-have", "no-whitespace", "use-dashes" ]
paper_authors: ["surname, name", "surname, name"]
paper_key: "bibtex_citekey"
```

Even though they can be extracted from the bibliography file, it is
necessary to copy the authors to `paper_authors`. This is required
for the "Papers by..." pages.

The name of the file should coincide with the bibtex
citekey. Eventually we will drop the additional variable in the
frontmatter.

## Editing the markdown directly

Simple but tedious: use markdown directly with,
e.g. [easy-hugo for emacs](https://github.com/masasam/emacs-easy-hugo). This
makes browsing the posts easy and copies a frontmatter template from
`archetypes/default.md`.

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
Branch `master` is automatically deployed onto [netlify](https://netlify.com).

## Credits

* The many authors of the papers
* HUGO
* hikari theme (port and original author)
* MathJax
* jQuery
* icomoon
* And of course **TeXmacs** :)
