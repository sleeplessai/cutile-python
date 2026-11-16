# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= -W
VENV          ?=
SPHINXBUILD   ?= $(if $(VENV),$(VENV)/bin/,)sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
PROJECTDIR    = $(CURDIR)/..
LANGUAGES     ?= en zh_CN

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean gettext update-po html-i18n serve serve-i18n

clean:
	rm -rf $(SOURCEDIR)/generated
	rm -rf $(BUILDDIR)

gettext:
	$(SPHINXBUILD) -b gettext "$(SOURCEDIR)" "$(BUILDDIR)/gettext" $(O) --show-traceback

update-po: gettext
	$(if $(VENV),$(VENV)/bin/,)sphinx-intl update -p "$(BUILDDIR)/gettext" $(foreach l,$(LANGUAGES),-l $(l)) -d "$(SOURCEDIR)/locale"

html-i18n:
	@for lang in $(LANGUAGES); do \
		$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)/i18n_tmp/$$lang" -D language=$$lang $(SPHINXOPTS) $(O) --show-traceback; \
		mkdir -p "$(BUILDDIR)/html/$$lang"; \
		cp -r "$(BUILDDIR)/i18n_tmp/$$lang/html/." "$(BUILDDIR)/html/$$lang/"; \
	done
	cp "$(SOURCEDIR)/_templates/root-redirect.html" "$(BUILDDIR)/html/index.html"

serve: html
	$(if $(VENV),$(VENV)/bin/,)python -m http.server -d "$(BUILDDIR)/html"

serve-i18n: html-i18n
	$(if $(VENV),$(VENV)/bin/,)python -m http.server -d "$(BUILDDIR)/html"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option. $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) --show-traceback
