import { ImageList, ImageListItem, ImageListItemBar } from '@mui/material';
import { FormControl, Grid, InputLabel, MenuItem, Select, Typography } from '@mui/material';
import React from 'react';


const imgStyle = {
  borderRadius: 20,
};

const colsStyle = {
  width: 75,
};

export default function DetectionGallery(props) {
  const [cols, setCols] = React.useState(2);

  const handleColsChange = (event) => {
    setCols(event.target.value);
  };

  if (props.detectionItems.length > 0) {
    return (
      <Grid item xs={12}>
        <Grid container>
          <Grid item xs={9}>
            <Typography variant="h4" align="left">Detections</Typography>
          </Grid>

          <Grid item xs={3} textAlign="end">
            <FormControl style={colsStyle}>
              <InputLabel id="columns-select-label">Columns</InputLabel>
              <Select
                labelId="columns-select-label"
                value={cols}
                label="Columns"
                onChange={handleColsChange}
              >
                <MenuItem value={1}>1</MenuItem>
                <MenuItem value={2}>2</MenuItem>
                <MenuItem value={3}>3</MenuItem>
              </Select>
            </FormControl>
          </Grid>

          <Grid item xs={12}>
            <ImageList cols={cols} gap={12}>
              {props.detectionItems.map(
                ({username, src, logos, originalSrc, likes, timestamp}, i) => (
                <ImageListItem key={i}>
                  <img src={src} alt={'photo ' + i} loading="lazy" style={imgStyle} />
                  <ImageListItemBar
                    title={<a href={originalSrc} rel="noreferrer" target="_blank">Original post</a>}
                    subtitle={
                      <div>
                        <p>Logos: {logos.join(', ')}</p>
                        <p>Username: {username}</p>
                        <p>Likes: {likes}</p>
                        <p>Post date: {new Date(timestamp).toLocaleDateString()}</p>
                      </div>
                    }
                    position="below"
                  />
                </ImageListItem>
              ))}
            </ImageList>
          </Grid>
        </Grid>
      </Grid>
    );
  }
};
