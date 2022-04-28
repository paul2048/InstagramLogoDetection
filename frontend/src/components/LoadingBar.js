import { Box, LinearProgress, Grid, Typography } from '@mui/material';
import React from 'react';
import io from 'socket.io-client';


const socket = io.connect('http://127.0.0.1:5000');

const loadingBarStyle = {
  borderRadius: 40,
  height: 16,
}

export default function LoadingBar({ username, detectionItems }) {
  const [progress, setProgress] = React.useState('0.00');

  // Update the percentage progress for the current username's loading bar
  socket.on(`send_progress_${username}`, ({ progress }) => {
    setProgress(progress);
  });

  // When the `detectionItems` state gets updated, set the progress of newly
  // added username to "0".
  React.useEffect(() => {
    if (detectionItems[username].length === 0) {
      setProgress(0);
    }
  }, [detectionItems]);

  return (
    <Grid item xs={12}>
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <Box sx={{ width: '100%', mr: 1 }}>
          <LinearProgress variant="determinate" value={+progress} style={loadingBarStyle} />
        </Box>
        <Box sx={{ minWidth: 35 }}>
          <Typography variant="body2" color="text.secondary">{progress}%</Typography>
        </Box>
      </Box>
    </Grid>
  );
}
