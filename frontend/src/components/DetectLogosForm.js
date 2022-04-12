import React from 'react';
import { FormGroup, Grid } from '@mui/material';
import { Autocomplete, Chip, TextField } from '@mui/material';
import { LoadingButton } from '@mui/lab';
import PlayCircleFilledRoundedIcon from '@mui/icons-material/PlayCircleFilledRounded';
import ChipInput from 'material-ui-chip-input'
import axios from 'axios';
import io from 'socket.io-client';
import DetectionGallery from './DetectionGallery';


const socket = io.connect('http://127.0.0.1:5000');

export default function DetectLogosForm() {
  const [usernames, setUsernames] = React.useState([]);
  const [logos, setLogos] = React.useState([]);
  // const [inputLogos, setInputLogos] = React.useState([]);
  const [open, setOpen] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const [detectionItems, setDetectionItems] = React.useState([]);
  const loadingLogos = (open && logos.length === 0);

  const handleAddAccount = (account) => {
    setUsernames([...usernames, account]);
  }

  const handleRemoveAccount = (account, index) => {
    const newAcccounts = usernames.filter((_, i) => i !== index);
    setUsernames(newAcccounts);
  }

  const handleClick = () => {
    setLoading(!loading);
    axios.post('http://127.0.0.1:5000/detect_logos', {logos, usernames})
      .then(() => {
        setLoading(false);
      })
      .catch((error) => {
        console.error(error);
      });
  }

  const getLogos = () => {
    axios.get('http://127.0.0.1:5000/get_logos')
      .catch((error) => {
        console.error(error);
      });
  };

  socket.on('receive_detection_image', (detectionItem) => {
    setDetectionItems([...detectionItems, detectionItem]);
  });

  React.useEffect(() => {
    let active = true;

    if (!loadingLogos) {
      return undefined;
    }

    (async () => {
      if (active) {
        setLogos((await getLogos()).data.logos);
      }
    })();

    return () => (active = false);
  }, [loadingLogos]);

  return (
    <FormGroup>
      <Grid container spacing={4}>
        <Grid item xs={12} md={6}>
          <ChipInput
            value={usernames}
            onAdd={(chip) => handleAddAccount(chip)}
            onDelete={(chip, index) => handleRemoveAccount(chip, index)}
            variant={'outlined'}
            // Older browsers return 'Spacebar' instead of ' ', so we include both
            newChipKeys={['Enter', ' ', 'Spacebar', 'Tab']}
            fullWidth
            label={'Instagram usernames'}
            placeholder={'Instagram usernames'}
          />
        </Grid>

        <Grid item xs={12} md={6}>
          <Autocomplete
            multiple
            fullWidth
            open={open}
            onOpen={() => {
              setOpen(true);
            }}
            onClose={() => {
              setOpen(false);
            }}
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
          />
        </Grid>

        <Grid item xs={12}>
          <LoadingButton
            loading={loading}
            loadingPosition="start"
            onClick={handleClick}
            startIcon={<PlayCircleFilledRoundedIcon />}
            variant="contained"
            size="large"
          >
            Detect Logos
          </LoadingButton>
        </Grid>

        <DetectionGallery detectionItems={detectionItems} />
      </Grid>
    </FormGroup>
  );
};
