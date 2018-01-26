package de.mpicbg.mosaic.apr.tests;


import de.mpicbg.mosaic.apr.APRIteratorStd;
import de.mpicbg.mosaic.apr.APRStd;
import de.mpicbg.mosaic.apr.ExtraParticleDataFloat;
import de.mpicbg.mosaic.apr.Loader;
import org.junit.Test;

import java.io.IOException;
import java.math.BigInteger;

/**
 * <Description>
 *
 * @author Ulrik Günther <hello@ulrik.is>
 */
public class TestAPRIterate {
  static {
    try {
      Loader.loadNatives();
    } catch (IOException e) {
      System.err.println("Could not load natives.");
      e.printStackTrace();
    }

    System.err.println("PID=" + getPID());
  }

  public static long getPID() {
    String processName = java.lang.management.ManagementFactory.getRuntimeMXBean().getName();
    if (processName != null && processName.length() > 0) {
      try {
        return Long.parseLong(processName.split("@")[0]);
      }
      catch (Exception e) {
        return 0;
      }
    }

    return 0;
  }


  @Test
  public void loopSerial() {
    System.out.println("Serial Looping");
    final APRStd apr = new APRStd();

    final String path = System.getProperty("apr.testfile", this.getClass().getResource("sphere.h5").getPath());

    System.out.println("Loading APR file from " + path + " ...");

    apr.read_apr(path);

    System.out.println("Loaded file, got " + apr.total_number_particles() + " particles.\n");

    final APRIteratorStd aprIterator = new APRIteratorStd(apr);
    final ExtraParticleDataFloat extraParticleData = new ExtraParticleDataFloat(apr);

    assert(extraParticleData.total_number_particles().longValue() == apr.total_number_particles().longValue());

    final long showParticleDivisor = apr.total_number_particles().longValue()/10;

    System.out.println("Example particles: ");

    for(long particleNumber = 0; particleNumber < apr.total_number_particles().longValue(); particleNumber++) {
      aprIterator.set_iterator_to_particle_by_number(BigInteger.valueOf(particleNumber));

      int spatialX = aprIterator.x();
      int spatialY = aprIterator.y();
      int spatialZ = aprIterator.z();

      int level = aprIterator.level();
      int type = aprIterator.type();

      extraParticleData.set_particle(aprIterator, apr.getParticles_intensities().get_particle(aprIterator));
      float intensity = extraParticleData.get_particle(aprIterator);

      if(particleNumber % showParticleDivisor == 0) {
        System.out.println(String.format("  %d: %d %d %d: level=%d, type=%d, intensity=%f",
                particleNumber, spatialX, spatialY, spatialZ, level, type, intensity));
      }
    }
  }

  @Test
  public void loopSerialByLevel() {
    System.out.println("Serial Looping by level");
    final APRStd apr = new APRStd();

    final String path = System.getProperty("apr.testfile", this.getClass().getResource("sphere.h5").getPath());

    System.out.println("Loading APR file from " + path + " ...");

    apr.read_apr(path);

    System.out.println("Loaded file, got " + apr.total_number_particles() + " particles.\n");

    final APRIteratorStd aprIterator = new APRIteratorStd(apr);
    final ExtraParticleDataFloat extraParticleData = new ExtraParticleDataFloat(apr);

    assert(extraParticleData.total_number_particles().longValue() == apr.total_number_particles().longValue());

    final long showParticleDivisor = apr.total_number_particles().longValue()/10;

    System.out.println("Example particles: ");

    long particleNumber = 0l;
    for(int level = aprIterator.level_min(); level <= aprIterator.level_max(); ++level) {
      for(particleNumber = aprIterator.particles_level_begin(level).longValue(); particleNumber < aprIterator.particles_level_end(level).longValue(); ++particleNumber) {
        aprIterator.set_iterator_to_particle_by_number(BigInteger.valueOf(particleNumber));

        int spatialX = aprIterator.x();
        int spatialY = aprIterator.y();
        int spatialZ = aprIterator.z();

        long closestPixelX = aprIterator.x_nearest_pixel();
        long closestPixelY = aprIterator.y_nearest_pixel();
        long closestPixelZ = aprIterator.z_nearest_pixel();

        if(particleNumber % showParticleDivisor == 0) {
          System.out.println(String.format(" %d, nearest pixel: %d %d %d: level=%d",
                  particleNumber, closestPixelX, closestPixelY, closestPixelZ, level));
        }
      }
    }
  }
}
