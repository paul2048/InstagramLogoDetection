import { Accordion, AccordionDetails, AccordionSummary, Grow } from '@mui/material';
import { ImageList, ImageListItem, ImageListItemBar } from '@mui/material';
import { FormControl, Grid, InputLabel, MenuItem, Select, Typography } from '@mui/material';
import LoadingBar from './LoadingBar';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import React from 'react';


const imgStyle = {
  borderRadius: 20,
};

const colsStyle = {
  width: 75,
};

export default function DetectionGallery({ detectionItems }) {
  const [cols, setCols] = React.useState(2);

  const handleColsChange = (event) => {
    setCols(event.target.value);
  };

  return (
    <Grid item xs={12}>
      <Grid container spacing={3}>
        <Grid item xs={9} alignSelf="center">
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
          {Object.keys(detectionItems).map((username, i) => (
            <Accordion key={i}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography>{username}</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <ImageList cols={cols} gap={12}>
                  {detectionItems[username].map(({src, logos, originalSrc, likes, timestamp}, j) => (
                    <Grow key={j} in={true}>
                      <ImageListItem>
                        <img src={src} alt={'photo ' + j} loading="lazy" style={imgStyle} />
                        <ImageListItemBar
                          title={<a href={originalSrc} rel="noreferrer" target="_blank">Original post</a>}
                          subtitle={
                            <div>
                              <p>Logos: {logos.join(', ')}</p>
                              <p>Likes: {likes}</p>
                              <p>Post date: {new Date(timestamp).toLocaleDateString()}</p>
                            </div>
                          }
                          position="below"
                        />
                      </ImageListItem>
                    </Grow>
                  ))}
                </ImageList>
                <LoadingBar username={username} detectionItems={detectionItems} />
              </AccordionDetails>
            </Accordion>
          ))}
        </Grid>
      </Grid>
    </Grid>
  );
};
