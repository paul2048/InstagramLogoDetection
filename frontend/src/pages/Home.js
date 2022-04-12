import React from 'react';
import Paper from '@mui/material/Paper';
import Grid from '@mui/material/Grid';
import Typography from '@mui/material/Typography';
import DetectLogosForm from '../components/DetectLogosForm';


const style = {
  backgroundImage: `url(${require('../images/wave.svg').default})`,
  backgroundSize: 'cover',
  backgroundPosition: 'center',
  color: '#212121',
};

export default function Home() {
  return (
    <Grid container spacing={4}>
      <Grid item xs={12}>
        <Paper style={style}>
          <Typography variant="h2" fontWeight={700}>Logo Det.</Typography>
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
