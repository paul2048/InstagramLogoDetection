import { Grid, ImageList, ImageListItem, ImageListItemBar, Typography } from '@mui/material';
import React from 'react';

export default function DetectionGallery(props) {
  const imgStyle = {
    borderRadius: 20,
  }

  if (props.detectionItems.length > 0) {
    return (
      <Grid item xs={12}>
        <Typography variant='h3' align='left'>Detections</Typography>
        <ImageList>
          {props.detectionItems.map(({username, src, logos}, i) => (
            <ImageListItem key={i}>
              <img src={src} alt={'photo ' + i} loading="lazy" style={imgStyle} />
              <ImageListItemBar
                title={'title'}
                subtitle={<span>For {username} found: {logos}</span>}
                position="below"
              />
            </ImageListItem>
          ))}
        </ImageList>
      </Grid>
    );
  }
};
