autoflake --remove-all-unused-imports --in-place -r --exclude=notebooks/outputs --exclude=templates/outputs --exclude=training/outputs ./
isort --skip notebooks/outputs --skip templates/outputs --skip training/outputs ./
black --exclude 'notebooks/outputs|templates/outputs|training/outputs' ./