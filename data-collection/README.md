# Server

This server will serve the build webpage found at `dist/index.html` and collect
answers storing them in `form.json`.

To serve the data collection form first ensure all Python dependencies are
installed, they are listed in `requirements.txt` in the root directory:

`python3 install -r requirements.txt`

Then to start the server: `python3 server.py`

Host and port settings can be found in the `__main__` of `server.py`.


# Developing

If you want to develop the form you first need to install the tool
[yarn](https://yarnpkg.com/en/).

Then to install dependencies run `yarn` in this folder.

To develop with "hot reloading": `yarn dev`

To rebuild the webpage: `yarn build`
