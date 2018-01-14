import Auditorium from 'src/Auditorium.jsx';
import React from 'react';
import request from 'superagent';

// Should be rectangular!
const SEATS = [
  [0, 1, 1, 0, 1, 1, 0],
  [0, 1, 1, 0, 1, 1, 0],
  [0, 1, 1, 0, 1, 1, 0],
  [0, 1, 1, 0, 1, 1, 0],
];

// Style for the button container.
const buttonContainerStyle = {
  textAlign: 'center',
};

// Style for the button.
const buttonStyle = {
  backgroundColor: 'white',
  borderRadius: '3px',
  fontSize: 32,
};

class Form extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      done: false,
      timestamp: new Date().getTime(),
    };
  }

  // Update local state with a user's answer.
  handleInput = (input, key) => {
    this.state[key] = input;
  }

  // Show thank you message and send data to server.
  handleDone = () => {
    this.setState({ done: true });
    request.post('/done').send(this.state).then(console.log);
  }

  // Render a a given question.
  renderQuestion(question, questionID) {
    return (
      <div>
        <div>{question}</div>
        <input
          type="text"
          onChange={e => this.handleInput(e.target.value, questionID)}
        />
      </div>
    );
  }

  render() {
    if (this.state.done) {
      return <div>THANK YOU!</div>;
    }
    return (
      <div className="form">
        {this.renderQuestion('Enter your given ID', 'userid')}
        {this.renderQuestion(`If there is a person sitting
           immediately to your left, how well do you know them
           on a 0-10 scale, 0 being not at all, 10 being your
           best friend`, 'left')
        }
        {this.renderQuestion(`If there is a person sitting
           immediately to your right, how well do you know them
           on a 0-10 scale, 0 being not at all, 10 being your
           best friend`, 'right')
        }
        <Auditorium
          question="Where are you seated?"
          seats={SEATS}
          lift={seat => this.handleInput(seat, 'loc')}
        />
        <Auditorium
          question="Disregarding where others are, where is your preferred seat?"
          seats={SEATS}
          lift={seat => this.handleInput(seat, 'pref')}
        />
        <div>Please ensure answers are correct :)</div>
        <div style={buttonContainerStyle}>
          <button style={buttonStyle} onClick={this.handleDone}>SEND</button>
        </div>
      </div>
    );
  }
}

export default Form;
