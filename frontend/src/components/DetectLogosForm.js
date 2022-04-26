import React from 'react';
import { FormGroup, Grid, Autocomplete, Chip, TextField } from '@mui/material';
import { LoadingButton } from '@mui/lab';
import PlayCircleFilledRoundedIcon from '@mui/icons-material/PlayCircleFilledRounded';
import ChipInput from 'material-ui-chip-input'
import axios from 'axios';
import io from 'socket.io-client';
import DetectionGallery from './DetectionGallery';


const socket = io.connect('http://127.0.0.1:5000');

export default function DetectLogosForm() {
  const [usernames, setUsernames] = React.useState(['mkbhd', 'everythingapplepro']);
  const [logos, setLogos] = React.useState([]);
  const [selectedLogos, setSelectedLogos] = React.useState(['Apple', 'Asus']);
  const [open, setOpen] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const [detectionItems, setDetectionItems] = React.useState({});
  const loadingSelectableLogos = (open && logos.length === 0);

  const handleAddUsername = (account) => {
    setUsernames([...usernames, account]);
    setDetectionItems(Object.fromEntries(usernames.map((key) => [key, []])))
  }

  const handleRemoveUsername = (account, index) => {
    // Filter out the i-th account (the account of the clicked 'X')
    const newAcccounts = usernames.filter((_, i) => (i !== index));
    setUsernames(newAcccounts);
    setDetectionItems(Object.fromEntries(usernames.map((key) => [key, []])))
  }

  const handleDetectLogosClick = () => {
    setLoading(!loading);
    // Send the selected usernames and logos, so the logo detection process starts
    axios.post('http://127.0.0.1:5000/detect_logos', {usernames, selectedLogos})
      .then(() => {
        setLoading(false);
      })
      .catch((error) => {
        console.error(error);
      });
  }

  socket.on('receive_detection_image', (detectionItem) => {
    const username = detectionItem.username;
    let userArr = [detectionItem]
    if (username in detectionItems) {
      userArr = [...(detectionItems[username]), detectionItem]
    }
    // Append the detection item to the corresponding instagram username key
    setDetectionItems({
      ...detectionItems,
      [username]: userArr
    });
  });

  React.useEffect(() => {
    let active = true;

    if (!loadingSelectableLogos) {
      return undefined;
    }

    // If the select input is active, fill the select input with the possible logos
    (async () => {
      if (active) {
        const possibleLogos = await axios.get('http://127.0.0.1:5000/get_logos')
          .catch((error) => {
            console.error(error);
          });
        setLogos((possibleLogos).data.logos);
      }
    })();

    return () => (active = false);
  }, [loadingSelectableLogos]);

  React.useEffect(() => {
    // Each username will map to an empty array of detections
    setDetectionItems(Object.fromEntries(usernames.map((key) => [key, []])));
  }, [usernames]);

  return (
    <FormGroup>
      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <ChipInput
            value={usernames}
            onAdd={(chip) => handleAddUsername(chip)}
            onDelete={(chip, index) => handleRemoveUsername(chip, index)}
            variant={'outlined'}
            // Older browsers return 'Spacebar' instead of ' ', so we include both
            newChipKeys={['Enter', ' ', 'Spacebar', 'Tab']}
            fullWidth
            label={'Instagram usernames'}
            placeholder={'Instagram usernames'}
            disabled={loading}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Autocomplete
            value={selectedLogos}
            multiple
            fullWidth
            open={open}
            onOpen={() => setOpen(true)}
            onChange={(_, val) => setSelectedLogos(val)}
            onClose={() => setOpen(false)}
            limitTags={64}
            options={logos}
            getOptionLabel={(option) => option}
            renderTags={(tagValue, getTagProps) =>
              tagValue.map((option, index) => (
                <Chip label={option} {...getTagProps({ index })} />
              ))
            }
            renderInput={(params) => (
              <TextField {...params} label="Logos" placeholder="Logos" />
            )}
            disabled={loading}
          />
        </Grid>

        <Grid item xs={12}>
          <LoadingButton
            loading={loading}
            loadingPosition="start"
            onClick={handleDetectLogosClick}
            startIcon={<PlayCircleFilledRoundedIcon />}
            variant="contained"
            size="large"
          >
            Detect Logos
          </LoadingButton>
        </Grid>

        {/* Only display the detections after the "Detect Logos" button was pressed */}
        {Object.keys(detectionItems).length > 0 &&
          <DetectionGallery detectionItems={detectionItems} />}
      </Grid>
    </FormGroup>
  );
};
