import PropTypes from 'prop-types';
import React from 'react';

const EMPTY = 0;
const SEAT = 1;

// Style for container which contains rows.
const auditoriumStyle = {
  display: 'flex',
  flexDirection: 'column',
  width: '100vw',
};

// Style for text above of below auditorium.
const infoTextStyle = {
  textAlign: 'center',
};

// Style for a row of seats.
const rowStyle = {
  display: 'flex',
};

// Style used by all cells.
const cellStyle = {
  flexGrow: 1,
  minHeight: '50px',
  border: '1px solid black',
};

// Style for an empty cell.
const emptyStyle = {
  ...cellStyle,
  backgroundColor: 'lightgray',
};

// Style for a cell with a seat.
const seatStyle = {
  ...cellStyle,
  backgroundColor: 'lightblue',
};

// Style for a selected cell.
const selectedStyle = {
  ...cellStyle,
  backgroundColor: 'red',
};

class Auditorium extends React.Component {
  static propTypes = {
    // The question asked about the auditorium.
    question: PropTypes.string.isRequired,
    // A 2d array of 0 (an empty cell) or 1 (a seat).
    seats: PropTypes.array.isRequired,
    // Function to return selected seat to parent.
    lift: PropTypes.func.isRequired,
  }

  constructor(props) {
    super(props);
    this.state = {
      // The currently selected seat, [i, j].
      // No seat initially selected.
      selected: [-1, -1],
    };
  }

  // Select a seat by given indices.
  selectSeat = (i, j) => {
    if (this.props.seats[i][j] === SEAT) {
      this.props.lift(`${i}, ${j}`);
      this.setState({ selected: [i, j] });
    }
  }

  render() {
    // First we build our rows of empty and seat cells.
    const height = this.props.seats.length;
    const width = this.props.seats[0].length;
    const rows = [];
    // For each row in Auditorium.
    for (let i = 0; i < height; i += 1) {
      const row = [];
      // For each column in row.
      for (let j = 0; j < width; j += 1) {
        let thisCellStyle = {};
        switch (this.props.seats[i][j]) {
          case EMPTY:
            thisCellStyle = emptyStyle;
            break;
          case SEAT:
            thisCellStyle = seatStyle;
            break;
          default: throw new Error('Unknown cell type');
        }
        // Override style if seat is selected.
        if (i === this.state.selected[0] &&
            j === this.state.selected[1]) {
          thisCellStyle = selectedStyle;
        }
        row.push(<div
          style={thisCellStyle}
          onClick={() => this.selectSeat(i, j)}
        />);
      }
      rows.push(<div style={rowStyle}>{row}</div>);
    }
    // Return the HTML structure with the above rows.
    return (
      <div>
        <div>{this.props.question}</div>
        <div style={auditoriumStyle}>
          <div style={infoTextStyle}>BACK WALL</div>
          {rows}
          <div style={infoTextStyle}>LECTURER</div>
        </div>
      </div>
    );
  }
}

export default Auditorium;
