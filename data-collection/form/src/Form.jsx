import { SEATS, TAKEN } from 'src/seats.jsx';
import Auditorium from 'src/Auditorium.jsx';
import React from 'react';
import Slider from 'src/Slider.jsx';
import request from 'superagent';

const buttonStyle = {
  backgroundColor: 'white',
  borderRadius: '3px',
  fontSize: 32,
};

const infoText = {
  color: 'slategray',
};

class Form extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      // Is the form filled in?
      done: false,
      // When the form is initially loaded.
      timestamp: new Date().getTime(),
    };
  }

  // Update local state with a user's answer.
  handleInput = (input, key) => {
    console.log(`Input: ${input} from key ${key}`);
    this.state[key] = input;
  }

  // Show thank you message and send data to server.
  handleDone = () => {
    this.setState({ done: true });
    request.post('/done').send(this.state).then(console.log);
  }

  // Render a a given question.
  renderQuestion(question, info, questionID) {
    return (
      <div>
        <div>{question}</div>
        {info ? <div style={infoText}>{info}</div> : null}
        <input
          type="number"
          onChange={e => this.handleInput(e.target.value, questionID)}
        />
      </div>
    );
  }

  // Render a discrete 1-5 question.
  render1To5(question, info, questionID, allow0) {
    return (
      <div>
        <div>{question}</div>
        <div style={infoText}>{info}</div>
        <input
          type="number"
          min={allow0 ? 0 : 1}
          max={5}
          step={1}
          onChange={e => this.handleInput(e.target.value, questionID)}
        />
      </div>
    );
  }

  render() {
    // If we're done we only show a thank you message.
    if (this.state.done) {
      return <div>THANK YOU!</div>;
    }

    return (
      <div className="form">

        <Auditorium
          question="Where are you seated?"
          seats={SEATS}
          lift={seat => this.handleInput(seat, 'seatlocation')}
        />

        {this.renderQuestion(
          `When you sat down in your seat initially, how many people did you
           have to pass in your row to get to your seat?`,
          null, 'crosscost',
        )}

        {this.renderQuestion(
          `Of the students attending this course, with how many people do you
           consider yourself familiar?`,
          '(Cambridge dictionary: familiar, to know something or someone well.)',
          'coursefriends',
        )}

        {this.render1To5(
          `If someone is sitting immediately to your left, how familiar are you
           with them?`,
          `(0 = Nobody immediately left. 1 = Not familiar. 2 = Slightly
           familiar. 3 = Somewhat familiar. 4 = Familiar. 5 = Very familiar.)`,
          'knowleft',
          true,
        )}

        {this.render1To5(
          `If someone is sitting immediately to your right, how familiar are
           you with them?`,
          `(0 = Nobody immediately right. 1 = Not familiar. 2 = Slightly
           familiar. 3 = Somewhat familiar. 4 = Familiar. 5 = Very familiar.)`,
          'knowright',
          true,
        )}

        {this.render1To5(
          `On entering a lecture room how important is it to you to sit next to
           someone with whom you are familiar?`,
          `(1 = Not important. 2 = Slightly important. 3 = Somewhat important. 4
           = Important. 5 = Very important.)`,
          'sitnexttofamiliar',
        )}

        {this.render1To5(
          `Consider entering a lecture room where you don't know anyone, how
           important to you is it to sit next to another person?`,
          `(1 = Not important. 2 = Slightly important. 3 = Somewhat important. 4
           = Important. 5 = Very important.)`,
          'sitnexttoperson',
        )}

        {this.render1To5(
          `On entering a lecture room how important is it to you to find a seat
           that is in a good physical location?`,
          `(1 = Not important. 2 = Slightly important. 3 = Somewhat important. 4
           = Important. 5 = Very important.)`,
          'sitgoodlocation',
        )}

        {this.render1To5(
          `On entering a lecture room how important is it to you to find a seat
           that is easy to get to?`,
          `(1 = Not important. 2 = Slightly important. 3 = Somewhat important. 4
           = Important. 5 = Very important.)`,
          'siteasyreach',
        )}

        {/* <Slider
            question="Estimate the ratio of importance to you of sitting
            next to a friend versus sitting in your preferred seat?"
            option_a="Next to a friend"
            option_b="In your preferred seat"
            lift={slider => this.handleInput(slider, 'slider')}
            /> */}

        <Auditorium
          question="Where is your preferred seat located if this is an empty
                   lecture theatre?"
          seats={SEATS}
          lift={seat => this.handleInput(seat, 'seatpreffered')}
        />

        <div>Please ensure answers are correct :)</div>
        <div style={{ textAlign: 'center' }}>
          <button style={buttonStyle} onClick={this.handleDone}>SEND</button>
        </div>
      </div>
    );
  }
}

export default Form;
