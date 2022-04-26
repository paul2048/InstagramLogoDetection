import { Box, LinearProgress, Grid, Typography } from '@mui/material';
import React from 'react';
import io from 'socket.io-client';


const socket = io.connect('http://127.0.0.1:5000');

export default function LoadingBar({ username }) {
  const [progress, setProgress] = React.useState(0.0);

  // Update the percentage progress for the current username's loading bar
  socket.on(`send_progress_${username}`, ({ progress }) => {
    setProgress(progress);
  });

  return (
    <Grid item xs={12}>
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <Box sx={{ width: '100%', mr: 1 }}>
          <LinearProgress variant="determinate" value={+progress} defaultValue={0.0} />
        </Box>
        <Box sx={{ minWidth: 35 }}>
          <Typography variant="body2" color="text.secondary">{progress}%</Typography>
        </Box>
      </Box>
    </Grid>
  );
}
