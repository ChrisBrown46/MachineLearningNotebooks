cd Notebooks
for %%f in (*.ipynb) do (
   jupyter-nbconvert --to html %%f
   del ..\docs\%%~nf.html
   move %%~nf.html ..\docs\%%~nf.html

   jupyter-nbconvert --to latex %%f
   del ..\Tex\%%~nf.tex
   move %%~nf.tex ..\Tex\%%~nf.tex

   jupyter-nbconvert --to pdf %%f
   del ..\Pdfs\%%~nf.pdf
   move %%~nf.pdf ..\Pdfs\%%~nf.pdf
)
cd ..

del /Q docs\assets\*
xcopy Notebooks\assets docs\assets

set /p UserInput="Commit message: "

git add .
git commit -m "%UserInput%"
git push
