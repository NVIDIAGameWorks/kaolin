# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -v
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
TYPEDOC       ?= npx typedoc

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile typedoc clean-typedoc

# Generate TypeDoc JSON before building Sphinx docs
typedoc:
	@echo "Generating TypeDoc JSON..."
	$(TYPEDOC) --options typedoc.json

# Clean TypeDoc output
clean-typedoc:
	rm -rf $(BUILDDIR)/typedoc $(BUILDDIR)/typedoc.json

# Override clean to also clean TypeDoc output
clean: clean-typedoc
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Override html target to run typedoc first
html: typedoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
