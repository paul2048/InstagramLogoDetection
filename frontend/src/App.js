import './App.css';
import Home from './pages/Home';
import React from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import Container from '@mui/material/Container';
import { CssBaseline } from '@mui/material';


const theme = createTheme({
  palette: {
    mode: 'dark',
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          padding: 50,
        },
      },
      defaultProps: {
        elevation: 8,
      },
    },
  },
  shape: {
    borderRadius: 50,
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg" className="App">
        <Home />
      </Container>
    </ThemeProvider>
  );
}

export default App;
