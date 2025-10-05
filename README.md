# Render App Base

It started as just doing one of the WGPU tutorials, but I'm trying to turn this into a project that can be used as a basis for any rendering apps I end up wanting to make.

## Ideals

This project becomes more useful, better organized, etc... But it should always strive for simplicity and easy understanding over "elegance".

Users should easily be able to find where code is and figure out how to use it, even if it means that I have to do some things a bit less efficiently or use more lines of code.

## Future Work

- [ ] Clean up and refactor rendering setup/config code away from the `app` and into the `render_lib` module.
- [ ] Track objects based on their layouts so that similar objects can be grouped together without modifying the whole render pipeline
- [ ] Move instancing into SSBO-based transform buffer and use vertex buffers for indexing rather than shoehorning transform matrices into vertex-like data.
