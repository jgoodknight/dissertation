name = google_abstract
blddir = _build

all: $(name).pdf

%.pdf: %.tex | $(blddir)
	@latexmk -halt-on-error -pdf -jobname=$(blddir)/$* $<
	@mv $(blddir)/$@ .

$(blddir):
	@mkdir $(blddir)

clean:
	rm -r $(blddir)

view:
	open -a Skim $(name).pdf

clear:
	rm -r $(blddir)
	rm $(name).pdf

remake:
	rm -r $(blddir)
	rm $(name).pdf
	$(name).pdf
	open -a Skim $(name).pdf
