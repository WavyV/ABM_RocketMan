import React from 'react';
import ReactDOM from 'react-dom';

class Form extends React.Component {

  constructor(props) {
    super(props);
  }

  handleInput = (event, ID) => {
    console.log(ID);
  }

  renderID() {
    return (
      <div>
        <div>Enter your given ID</div>
        <input type="text" onChange={e => this.handleInput(e, "ID")} />
      </div>
    );
  }

  render() {
    return (
      <div>
        {this.renderID()}
      </div>
    );
  }
}

export default Form;
