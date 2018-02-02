# Server

To serve the data collection form first ensure all Python dependencies are
installed, they are listed in `requirements.txt` in the root directory:

`python3 install -r requirements.txt`

Then to start the server: `python3 server.py`

Host and port settings can be found in the `__main__` of `server.py`.

All resulting answers will be found in this folder in `form.json`.

# Developing

If you want to develop the form you first need to install the tool
[https://yarnpkg.com/en/](Yarn).

Then to install dependencies run `yarn` in this folder.

To develop with "hot reloading": `yarn dev`

To rebuild the webpage: `yarn build`
