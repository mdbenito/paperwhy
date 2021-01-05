FROM alpine:3.12 AS build


ENV HUGO_VERSION=0.41

RUN apk add --no-cache curl 

# Install hugo
WORKDIR /
RUN curl -JLO https://github.com/gohugoio/hugo/releases/download/v${HUGO_VERSION}/hugo_${HUGO_VERSION}_Linux-64bit.tar.gz && \
    tar xzf hugo_${HUGO_VERSION}_Linux-64bit.tar.gz

FROM alpine:3.12

MAINTAINER Miguel de Benito <m.debenito.d@gmail.com>

RUN apk add --no-cache python2 python2-dev py-pip bash git

# Install pygments
RUN pip install --upgrade pip && \
    pip install Pygments==2.1.3 pygments-github-lexers==0.0.5

COPY --from=build /hugo /usr/bin/hugo

RUN mkdir /site
WORKDIR /site

ENTRYPOINT ["hugo"]
