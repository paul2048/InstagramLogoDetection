import React from 'react';
import Paper from '@mui/material/Paper';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography';
import DetectLogosForm from '../components/DetectLogosForm';


const paperStyle = {
  backgroundImage: `url(${require('../images/wave.svg').default})`,
  backgroundSize: 'cover',
  backgroundPosition: 'center',
  color: '#212121',
};

const h2Style = {
  fontWeight: 700,
  textShadow: '.5px .5px 2px #ccc',
}

export default function Home() {
  return (
    <Grid container spacing={4}>
      <Grid item xs={12}>
        <Paper style={paperStyle}>
          <Typography variant="h2" style={h2Style}>
            Instagram Logo Detection
          </Typography>
        </Paper>
      </Grid>

      <Grid item xs={12}>
        <Paper>
          <DetectLogosForm />
        </Paper>
      </Grid>
    </Grid>
  );
}
