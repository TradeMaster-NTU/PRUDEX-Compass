# Creating PRUDEX-Compass Using the Python Script
You can use the [`create_compass.py`](https://anonymous.4open.science/r/PRUDEX-Compass-68D5/Compass/generate/compass/create_compass.py) python script to generate a compass and specify how it is filled for each FinRL method in a JSON file:
```
$ python Compass/generate/compass/create_compass.py --h
usage: create_compass.py [-h] [--template TEMPLATE] [--output OUTPUT] [--data DATA]

CLEVA-Compass Generator.

optional arguments:
  -h, --help           show this help message and exit
  --template TEMPLATE  Tikz template file. (default: Compass/generate/compass/blank.tex)
  --output OUTPUT      Tikz filled output file. (default: Compass/generate/compass/filled.tex)
  --data DATA          Entries as JSON file. (default: Compass/generate/compass/data.json)
```
You can use the command 
```
$ python Compass/generate/compass/create_compass.py--data Compass/generate/compass/data.json --template Compass/generate/compass/blank_color.tex
``` 
to generate a more colorful picture
<div align="center">
  <img src="https://anonymous.4open.science/r/PRUDEX-Compass-948C/Compass/pictures/final/new_color.svg" width = 500 height = 400 />
</div>
or use the command 

```
$ python Compass/generate/compass/create_compass.py--data Compass/generate/compass/data.json
``` 
to generate the following picture.
<div align="center">
  <img src="https://anonymous.4open.science/r/PRUDEX-Compass-948C/Compass/pictures/final/old_color.svg" width = 500 height = 400 />
</div>



