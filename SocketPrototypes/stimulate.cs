using Mcs.Usb;

double amplitude_1, velocity_1;
double amplitude_2, velocity_2;

public class Stimulator
{
    private int Electrode1 = 0;
    private int Electrode2 = 1;
    private int HeadStage = 3;
    private void btStimulationStart_Click(object sender, EventArgs e)
        {
            int[] amplitude1 = new int[] { 10000, -10000, 0, 20000, -20000, 0 };
            int[] amplitude2 = new int[] { -10000, 10000, 0, -20000, 20000, 0 };
            int[] sideband1 = new int[] { 1 << 8, 3 << 8, 0, 1 << 8, 3 << 8, 0 }; // user defined sideband (use bits > 8)
            int[] StimulusActive1 = new int[] {1, 1, 0, 1, 1, 0};
            int[] sideband2 = new int[] { 4 << 8, 12 << 8, 0, 4 << 8, 12 << 8, 0 }; // user defined sideband (use bits > 8)
            int[] StimulusActive2 = new int[] { 1, 1, 0, 1, 1, 0 };
            ulong[] duration = new ulong[] {1000, 1000, 10000, 1000, 1000, 100000}; // could be different in length and numbers for each amplitude and sideband, it only needs to be equal in length for the individual amplitude and sideband 

            // Defining the Elektrodes. Choose if you want use automatic or manual mode for the stimulation electrode here:
            ElectrodeModeEnumNet electrodeMode = ElectrodeModeEnumNet.emAutomatic;
            //ElectrodeModeEnumNet electrodeMode = ElectrodeModeEnumNet.emManual;
            for (int i = 0; i < 60; i++)
            {
                stg.SetElectrodeMode((uint)HeadStage, (uint)i, i == Electrode1 || i == Electrode2 ? electrodeMode : ElectrodeModeEnumNet.emAutomatic);
                stg.SetElectrodeEnable((uint)HeadStage, (uint)i, 0, i == Electrode1 || i == Electrode2 ? true : false);
                stg.SetElectrodeDacMux((uint)HeadStage, (uint)i, 0, 
                    i == Electrode1 ? ElectrodeDacMuxEnumNet.Stg1 : 
                    ( i == Electrode2 ? ElectrodeDacMuxEnumNet.Stg2 : ElectrodeDacMuxEnumNet.Ground));
                stg.SetEnableAmplifierProtectionSwitch((uint)HeadStage, (uint)i, false); // Enable the switch if you want to protect the amplifier from an overload
                stg.SetBlankingEnable((uint)HeadStage, (uint)i, false); // Choose if you want Filter blanking during stimulation for an electrode
            }

            stg.SetVoltageMode(0);

            // bit0 (blanking switch) activation duration prolongation in µs
            uint Bit0Time = 40;

            // bit3 (stimulation switch) activation duration prolongation in µs
            uint Bit3Time = 800;

            // bit4 (stimulus selection switch) activation duration prolongation in µs
            uint Bit4Time = 40;

            // STG Channel 1:
            stg.PrepareAndSendData(2 * (uint)HeadStage + 0, amplitude1, duration, STG_DestinationEnumNet.channeldata_voltage);

            if (electrodeMode == ElectrodeModeEnumNet.emManual)
            {
                //pure user defined sideband:
                stg.PrepareAndSendData(2 * (uint)HeadStage + 0, sideband1, duration, STG_DestinationEnumNet.syncoutdata);
            }
            else
            {
                //alternative: adding sideband data for automatic stimulation mode:
                CStimulusFunctionNet.SidebandData SidebandData1 = stg.Stimulus.CreateSideband(StimulusActive1, sideband1, duration, Bit0Time, Bit3Time, Bit4Time);
                stg.PrepareAndSendData(2 * (uint) HeadStage + 0, SidebandData1.Sideband, SidebandData1.Duration, STG_DestinationEnumNet.syncoutdata);
            }

            // STG Channel 2:
            stg.PrepareAndSendData(2 * (uint)HeadStage + 1, amplitude2, duration, STG_DestinationEnumNet.channeldata_voltage);

            if (electrodeMode == ElectrodeModeEnumNet.emManual)
            {
                //pure user defined sideband:
                stg.PrepareAndSendData(2 * (uint)HeadStage + 1, sideband2, duration, STG_DestinationEnumNet.syncoutdata);
            }
            else
            {
                //alternative: adding sideband data for automatic stimulation mode:
                CStimulusFunctionNet.SidebandData SidebandData2 = stg.Stimulus.CreateSideband(StimulusActive2, sideband2, duration, Bit0Time, Bit3Time, Bit4Time);
                stg.PrepareAndSendData(2 * (uint) HeadStage + 1, SidebandData2.Sideband, SidebandData2.Duration, STG_DestinationEnumNet.syncoutdata);
            }

            stg.SetupTrigger(0, new uint[]{255}, new uint[]{255}, new uint[]{10});

            stg.SendStart(1);
        }

        private void btStimulatinStop_Click(object sender, EventArgs e)
        {
            stg.SendStop(1);
        }
}