import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [review, setReview] = useState('');
  const [sentiment, setSentiment] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://127.0.0.1:5000/predict', { review });
      setSentiment(response.data.sentiment);
    } catch (error) {
      console.error("There was an error making the request!", error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Sentiment Analysis</h1>
        <form onSubmit={handleSubmit}>
          <textarea
            value={review}
            onChange={(e) => setReview(e.target.value)}
            rows="4"
            cols="50"
            placeholder="Enter your review here..."
          />
          <br />
          <button type="submit">Analyze Sentiment</button>
        </form>
        {sentiment && <p>Sentiment: {sentiment}</p>}
      </header>
    </div>
  );
}

export default App;
