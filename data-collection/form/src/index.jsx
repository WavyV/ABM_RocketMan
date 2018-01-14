import Form from 'src/Form.jsx';
import React from 'react';
import ReactDOM from 'react-dom';

// Hot reload modules as they are updated.
if (module.hot) {
  module.hot.accept();
}

// Bind our form to the DOM.
ReactDOM.render(
  <Form />,
  document.getElementById('root'),
);
