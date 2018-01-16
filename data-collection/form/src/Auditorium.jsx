import PropTypes from 'prop-types';
import React from 'react';

const auditoriumStyle = {
  display: 'flex',
  flexDirection: 'column',
  width: '99vw',
};

const cellStyle = {
  flexGrow: 1,
  minHeight: '50px',
  border: '2px solid white',
  backgroundColor: 'lightgray',
};

class Auditorium extends React.Component {
  static propTypes = {
    // The question asked about the auditorium.
    question: PropTypes.string.isRequired,
    // A 2d array of 0 (an empty cell) or 1 (a seat).
    seats: PropTypes.array.isRequired,
    // A 2d array of 0 (unavailable) or 1 (taken).
    // Only used to make some seats unavailable.
    taken: PropTypes.array,
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
    // Can't select an unavailable seat.
    if (this.props.taken && this.props.taken[i][j]) {
      return;
    }
    // Select if the cell is a seat.
    if (this.props.seats[i][j] === 1) {
      this.props.lift(`${i}, ${j}`);
      this.setState({ selected: [i, j] });
    }
  }

  render() {
    // First we build our rows of empty and seat cells.
    const height = this.props.seats.length;
    const rows = [];
    // For each row in Auditorium.
    for (let i = 0; i < height; i += 1) {
      const row = [];
      // For each column in row.
      for (let j = 0; j < this.props.seats[i].length; j += 1) {
        const thisCellStyle = { ...cellStyle };
        // If a cell is a seat.
        if (this.props.seats[i][j]) {
          thisCellStyle.backgroundColor = 'orange';
        }
        // If a seat is unavailable.
        if (this.props.taken && this.props.taken[i][j]) {
          thisCellStyle.backgroundColor = 'black';
        }
        // If a cell is selected.
        if (i === this.state.selected[0] &&
            j === this.state.selected[1]) {
          thisCellStyle.backgroundColor = 'red';
        }
        // Add the seat to the row.
        row.push(<div
          style={thisCellStyle}
          onClick={() => this.selectSeat(i, j)}
        />);
      }
      // Add the row of seats to all rows.
      rows.push(<div style={{ display: 'flex' }}>{row}</div>);
    }

    // Return the HTML structure with the above rows.
    return (
      <div>
        <div>{this.props.question}</div>
        <div style={{ fontSize: 'large' }}>
          Answer by clicking on a seat (orange color).
        </div>
        <div style={{ fontSize: 'large' }}>
          {this.props.taken ? 'Black seats are taken' : null}
        </div>
        <div style={auditoriumStyle}>
          <div style={{ textAlign: 'center' }}>LECTURER</div>
          {rows}
          <div style={{ textAlign: 'center' }}>BACK WALL</div>
        </div>
      </div>
    );
  }
}

export default Auditorium;
