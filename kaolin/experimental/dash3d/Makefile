CORE_JS = $(wildcard src/*.js)
CORE_CSS = $(wildcard src/*.css)
THIRDPARTY_JS = ../../../node_modules/jquery/dist/jquery.min.js \
    ../../../node_modules/three/build/three.min.js \
    ../../../node_modules/three/examples/js/controls/OrbitControls.js   # TODO: this needs to be minified
THIRDPARTY_CSS = ../../../node_modules/mustard-ui/dist/css/mustard-ui.min.css
JSMIN=uglifyjs
AWK=awk

all: css js

js: jscore jsthirdparty

css: csscore cssthirdparty

info: info_jscore info_jsthirdparty info_csscore info_cssthirdparty

info_jscore: $(CORE_JS)
	echo "core js files: " $(CORE_JS)

info_jsthirdparty: $(THIRDPARTY_JS)
	echo "thirdparty js files: " $(THIRDPARTY_JS)

info_csscore: $(CORE_CSS)
	echo "core css files: " $(CORE_CSS)

info_cssthirdparty: $(THIRDPARTY_CSS)
	echo "thirdparty css files: " $(THIRDPARTY_CSS)

jscore: static/core-min.js

static/core-min.js: $(CORE_JS)
	cat src/copyright.js > static/core-min.js
	$(JSMIN) $^ -m >> static/core-min.js

jsthirdparty: static/thirdparty.js

static/thirdparty.js: $(THIRDPARTY_JS)
	$(AWK) 1 $^ > static/thirdparty.js

csscore: static/style.css

static/style.css: $(CORE_CSS)
	cat $^ > static/style.css

cssthirdparty: static/thirdparty.css

static/thirdparty.css: $(THIRDPARTY_CSS)
	cat $^ > static/thirdparty.css

clean:
	rm -f static/style.css static/core-min.js static/thirdparty.js static/thirdparty.css
