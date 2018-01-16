import PropTypes from 'prop-types';
import React from 'react';

class Slider extends React.Component {
  static propTypes = {
    // The question asked about the auditorium.
    question: PropTypes.string.isRequired,
    // The first option on the slider.
    option_a: PropTypes.string.isRequired,
    // The second option on the slider.
    option_b: PropTypes.string.isRequired,
    // The function to call on input change.
    lift: PropTypes.func.isRequired,
  }

  constructor(props) {
    super(props);
    this.state = {
      ratio_a: 50,
      ratio_b: 50,
    };
  }

  // Update ratios and call lift function.
  handleInput = (e) => {
    this.setState({
      ratio_a: e.target.value,
      ratio_b: 100 - e.target.value,
    });
    this.props.lift(e.target.value);
  }

  render() {
    return (
      <div>
        <div>{this.props.question}</div>
        <div style={{ fontSize: 'large' }}>
          {`${this.props.option_a} ${this.state.ratio_a}
           / ${this.state.ratio_b} ${this.props.option_b}`}
        </div>
        <input
          className="slider"
          type="range"
          min={0}
          max={100}
          step={1}
          onChange={this.handleInput}
        />
      </div>
    );
  }
}

export default Slider;
