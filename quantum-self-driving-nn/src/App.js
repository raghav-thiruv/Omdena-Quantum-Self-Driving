import logo from './logo.svg';
import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import NeuralNetworkPage from './components/NeuralNetworkPage';


function App() {
  return (
    <div className="App">
      <Router>
        <Routes>
          <Route path="/neural-network" element={<NeuralNetworkPage />} />
        </Routes>
      </Router>
    </div>
  );
}

export default App;
