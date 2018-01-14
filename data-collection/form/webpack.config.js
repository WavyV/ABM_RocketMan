const CleanWebpackPlugin = require('clean-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const path = require('path');
const { NamedModulesPlugin } = require('webpack');

// Code directories.
const dirs = {
  dist: path.join(__dirname, 'dist'),
  src: path.join(__dirname, 'src'),
};

module.exports = {
  // Entry point of our app.
  entry: path.join(dirs.src, 'index.jsx'),
  // Where to output bundled app.
  output: {
    filename: 'bundle.js',
    path: dirs.dist,
  },
  module: {
    rules: [
      // Load .jsx files using Babel.
      {
        test: /\.jsx$/,
        use: 'babel-loader',
      },
    ],
  },
  resolve: {
    alias: {
      src: dirs.src,
    },
  },
  plugins: [
    // Clean dist folder before build.
    new CleanWebpackPlugin([dirs.dist]),
    new HtmlWebpackPlugin({
      inject: 'body',
      template: path.join(dirs.src, 'index.html'),
    }),
    // Show names of hot reloaded modules.
    new NamedModulesPlugin(),
  ],
};
