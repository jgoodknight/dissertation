name = dissertation
blddir = _build

all: $(name).pdf

%.pdf: %.tex | $(blddir)
	@latexmk -halt-on-error -f -pdf -pvc -jobname=$(blddir)/$* $<
	@mv $(blddir)/$@ .

$(blddir):
	@mkdir $(blddir)

clean:
	rm -r $(blddir)
	rm $(name).pdf

view:
	open -a Skim $(name).pdf

spellcheck:
	find . -name '*.tex' -exec aspell --lang=en --mode=tex check "{}" \;
