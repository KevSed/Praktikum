all: python latex

python:
	cd analysis && mkdir plots && python auswertung.py

latex: FORCE | build
	@lualatex --halt-on-error --interaction=nonstopmode --output-directory=build --draftmode main.tex
	@biber build/main.bcf
	@lualatex --halt-on-error --interaction=nonstopmode --output-directory=build main.tex

FORCE:

build:
	@mkdir -p build/

clean:
	@rm -rf build/
